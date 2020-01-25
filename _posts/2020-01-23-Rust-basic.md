---
layout: post
title: "Rust basic"
categories: Rust, PL
author: lee gunjun
---

[Rust the book](https://doc.rust-lang.org/book/) 의 요약본입니다.

# Chapter 10. Generic Types, Traits, Lifetimes

<details>

## Generic

### Function

```
fn largest<T>(list: &[T]) -> T {
    let mut largest = list[0];

    for &item in list.iter() { // item 앞의 & 는 ref 를 벗겨내는 역할
        if item > largest { // -> Error
            largest = item;
        }
    }

    largest
}

fn main() {
    let numbers = vec![34, 50, 25, 100, 65];

    let result = largest(&numbers);
    println!("The largest number is {}", result);

    let chars = vec!['y', 'm', 'a', 'q'];

    let result = largest(&chars);
    println!("The largest char is {}", result);
}
```

place type name declarations inside angle brackets, <>, between the name of the function and the parameter list

참고: vector 를 argument 로 받을 때 Type 은 &[i32] 요렇게 함.

위 코드는 Error 가 발생함. 비교 연산을 가능케 하는 std::cmp::PartialOrd Trait 가 T Type 에 구현되어있지 않을 수 있기 때문임. 이에 대한 해결은 뒤에서 다시 보자. 

### Struct

struct 도 generic type parameter 사용 가능

```
struct Point<T> {
    x: T,
    y: T,
}
```

이렇게 하면

```
let first_point = Point{x: 1.0, y: 1.0};
let second_point = Point{x: 1.0, y: 1};
```

에서 첫 statement 은 컴파일이 되지만, 아래 statement 에서 컴파일 에러가 발생하는데 이는 두 타입이 다르기 때문.

```
struct Point<T, U> {
    x: T,
    y: U,
}
```

를 사용하여 해결 가능. 하지만 generic type 을 많이 사용하는 건 보통 코드를 더럽게 만들기 때문에 최대한 줄이자.

method 는 아래와 같이 짠다

```
impl<T> Point<T> {
    fn x(&self) -> &T {
        &self.x
    }
}
```

또는 specific type 에서만 동작하는 method 를 아래와 같이 짤 수도 있다.

```
impl Point<f32> {
    fn distance_from_origin(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2)).sqrt()
    }
}
```

T 가 f32 가 아닌 Point 들은 위의 method 를 가지지 못한다.

Generic type parameters in a struct definition aren’t always the same as those you use in that struct’s method signatures

```
struct Point<T, U> {
    x: T,
    y: U,
}

impl<T, U> Point<T, U> {
    fn mixup<V, W>(self, other: Point<V, W>) -> Point<T, W> {
        Point {
            x: self.x,
            y: other.y,
        }
    }
}
```

### Enum

enum Option<T> {
    Some(T),
    None,
}

enum Result<T, E> {
    Ok(T),
    Err(E),
}

Generic 을 사용해도 코드의 performance 저하는 전혀 없다. compile time 에 모든 generic type 을 concrete type 으로 바꾸는 Monomorphization 라 불리는 작업을 수행하기 때문.


## Trait

Trait 는 어떤 functionality 의 구현을 여러 타입들에게 강제하는 장치이다. 그리고 이를 통해 코드 중복을 없앨 수 있다.

두개의 Struct type: Newpaper, Tweet 를 가정하자.  
이 두 Struct 가 모두 summarize method 를 가진다하자

```
pub trait Summary {
    fn summarize(&self) -> String;
}
```

trait 의 Name 은 Summary 이고, 그 안에 method signature 를 정의했다.

이 trait 를 구현하는 모든 type 은 summarize 의 body 를 구현하는 코드를 가져야한다.

```
pub struct NewsArticle {
    pub headline: String,
    pub location: String,
    pub author: String,
    pub content: String,
}

impl Summary for NewsArticle {
    fn summarize(&self) -> String {
        format!("{}, by {} ({})", self.headline, self.author, self.location)
    }
}

pub struct Tweet {
    pub username: String,
    pub content: String,
    pub reply: bool,
    pub retweet: bool,
}

impl Summary for Tweet {
    fn summarize(&self) -> String {
        format!("{}: {}", self.username, self.content)
    }
}
```

한가지 주목해야하는 점은, trait 과 type 둘 중 하나라도 우리 crate local 에 있을때에만 그 type 에 그 trait 를 구현할 수 있다는 점입니다.  
즉 외부 라이브러리의 type 에 대해 외부 라이브러리의 trait 를 구현할 수 없습니다.  
예를 들어 Vec<T> 에 대한 Display trait 은 구현 불가능합니다.

참고로 

```
pub trait Summary {
    fn summarize(&self) -> String{
        String::from("(Read more...)")
    }
}
```

과 같이 trait 를 선언할 때 method 의 body 를 구현해줄수도 있습니다.  
이를 default implementation 이라 하며, trait 를 구현하는 type 에서 해당 method 를 구현하지 않았을때 default 로 행동하게 됩니다.

이제 이 trait 를 이용하여 코드 중복을 없애는 코드를 짜보자.

```
pub fn notify(item: impl Summary) {
    println!("Breaking news! {}", item.summarize());
}
```

와 같은 method 를 통해 코드 중복을 없앨 수 있다.

만약 trait 가 없었다면 각 type 별로 summarize 를 구현하고 notify 또한 구현해야 했을테지만, 이제는 summarize 만 각 type 별로 구현하고 notify 는 하나만 구현한뒤 Summary trait 를 구현하는 모든 타입에 대해 공유하도록 하면 된다.

```
pub fn notify<T: Summary>(item: T) {
    println!("Breaking news! {}", item.summarize());
}
```

똑같은 코드다. 하지만 더 간결하다. 이를 trait bounds 라고 부른다.

아래와 같은 코드를 통해 여러 trait 를 구현하는 타입에 대해 method 가 정의되도록 제한할수도 있다.

```
pub fn notify(item: impl Summary + Display) {
```

혹은

```
pub fn notify<T: Summary + Display>(item: T) {
```

trait bounds 를 사용해도 코드가 복잡해질 수 있다. 이는 where 를 통해 해결 가능하다. 

```
fn some_function<T: Display + Clone, U: Clone + Debug>(t: T, u: U) -> i32 {
```

위와 같은 복잡한 코드를 

```
fn some_function<T, U>(t: T, u: U) -> i32
    where T: Display + Clone,
          U: Clone + Debug
{
```

로 바꾸면 간결하지 않은가?

특정 trait 를 구현하는 type 을 return 하는 함수를 짤 수도 있다.

```
fn returns_summarizable() -> impl Summary {
    ...
}
```

위와 같은 method 는 Summary trait 를 구현하는 type 이면 전부 return 가능하다.

이제 앞에서 발생했던 largest 함수의 에러를 수정해보자.

```
fn largest<T: PartialOrd + Copy>(list: &[T]) -> T {
```

위와 같이 수정하면 성공이다. PartialOrd 는 크기비교 연산을 위해, Copy 는 `let mut largest = list[0];` statement 가 가능토록 만들기 위해 사용됐다.

## Lifetime

rust 는 dangling reference 를 방지하기 위해 lifetime 을 complie time 에 체크합니다.

아래의 코드가 dangling referene 에러를 유도하는 코드입니다.

```
{
    let r;

    {
        let x = 5;
        r = &x;
    }

    println!("r: {}", r);
}
```

compiler 는 borrow checker 를 통해 lifetime 을 체크합니다.

```
{
    let r;                // ---------+-- 'a
                          //          |
    {                     //          |
        let x = 5;        // -+-- 'b  |
        r = &x;           //  |       |
    }                     // -+       |
                          //          |
    println!("r: {}", r); //          |
}                         // ---------+
```

위 그림에서 lifetime 'b 가 'a 보다 빨리 끝나기 때문에 문제가 생긺을 쉽게 확인할 수 있습니다.

여기까지는 쉽죠. 그런데 아래와 같은 상황을 생각해봅시다.

```
fn main() {
    let string1 = String::from("abcd");
    let string2 = "xyz";

    let result = longest(string1.as_str(), string2);
    println!("The longest string is {}", result);
}

fn longest(x: &str, y: &str) -> &str { // -> Error!
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

위 코드는 컴파일 에러가 발생하는데 lifetime 을 컴파일러가 알수가 없기 때문입니다.

longest 함수가 x 를 리턴하냐 혹은 y 를 리턴하냐에 따라 return value 의 lifetime 이 다를 것입니다.  
런타임에 lifetime 때문에 오류가 생기는 것을 막고자 rust 는 컴파일 시간에 lifetime 문제를 해결하고자 합니다.  
그래서 우리는 longest method 의 lifetime 을 직접 지정해줄 필요가 있습니다.

```
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

위와 같이 lifetime 을 명시해주면 x 와 y 의 lifetime 의 교집합으로 lifetime 이 결정됩니다.

### lifetime in struct

```
struct ImportantExcerpt<'a> {
    part: &'a str,
}

impl<'a> ImportantExcerpt<'a> {
    fn level(&self) -> i32 {
        3
    }
    fn announce_and_return_part(&self, announcement: &str) -> &str {
        println!("Attention please: {}", announcement);
        self.part
    }
}
```

### Lifetime Elision

rust 의 모든 reference 는 lifetime 을 가집니다. 그런데 우리는 지금까지 lifetime 를 명시해주지 않았는데도 컴파일은 성공적이었습니다. 그 이유는 컴파일러가 lifetime 을 유추해내기 때문입니다.

lifetime 을 유추하는 3가지 rule 이 있습니다. 이는 공식문서를 참조해주세요.

### Static Lifetime

static lifetime `'static` 가 명시된 변수들은 프로그램의 entire duration 의 lifetime 을 가진다. 모든 string literal 의 lifetime 은 '`static' 이다


## Summary

```
use std::fmt::Display;

fn longest_with_an_announcement<'a, T>(x: &'a str, y: &'a str, ann: T) -> &'a str
    where T: Display
{
    println!("Announcement! {}", ann);
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
```

</details>


# Chapter 13. Iterators and Closures

<details>

## Clouser

Rust 의 closure 는 변수를 전달, 다른 함수에 argument 로 전달이 가능한 anonymous function 이다.

함수와 다르게 closure 는 정의된 scope 의 변수를 capture 할 수 있다.

```
let add_one = |x| {
    x + 1
};
```

위 closure 은 아래와 같이 축약 가능

```
let add_one = |x| x + 1;
```

### Closure Type Inference and Annotation

closure 는 타입 명시해줄 필요 없음. 아래 처럼 가능하긴 함

```
let add_one = |x: i32| -> i32 {
    x + 1
};
```

타입은 closure 가 처음 실행 되는 statement 에서 결정됨. 즉 아래와 같은 상황에선 Compile error 발생

```
let example_closure = |x| x;

let s = example_closure(String::from("hello"));
let n = example_closure(5);
```

### lazy evaluation with `Fn` traits

모든 closure 은 `Fn`, `FnOnce`, `FnMut` 중 적어도 하나의 trait 을 구현함. 이들에 대해선 나중에 다룸.

일단 코드부터 보자.

```
struct Cacher<T>
    where T: Fn(u32) -> u32
{
    calculation: T,
    value: Option<u32>,
}

impl<T> Cacher<T>
    where T: Fn(u32) -> u32
{
    fn new(calculation: T) -> Cacher<T> {
        Cacher {
            calculation,
            value: None,
        }
    }

    fn value(&mut self, arg: u32) -> u32 {
        match self.value {
            Some(v) => v,
            None => {
                let v = (self.calculation)(arg);
                self.value = Some(v);
                v
            },
        }
    }
}
```

근데 실제 코딩에서 위와 같은 코드는 피해야 함

```
#[test]
fn call_with_different_values() {
    let mut c = Cacher::new(|a| a);

    let v1 = c.value(1);
    let v2 = c.value(2);

    assert_eq!(v2, 2);
}
```

이런 상황 생기면 망하니까. HashMap 이용해서 해결 가능.

### Capturing the Environments with Closures

다음 코드는 가능함. 

```
fn main() {
    let x = 4;

    let equal_to_x = |z| z == x;

    let y = 4;

    assert!(equal_to_x(y));
}
```

다음 코드는 불가능함. function 은 dynamic environment 를 capture 할 수 없기 때문.

```
fn main() {
    let x = 4;

    fn equal_to_x(z: i32) -> bool { z == x }

    let y = 4;

    assert!(equal_to_x(y));
}
```

closure 는 environment 를 capture 하기 위한 memory 를 지정하는 overhead 를 기꺼이 감수해내지만, function 은 그런 overhead 가 생기는걸 용서못함.

Closures 가 environment 의 value 를 capture 하는 방식에는 총 3가지가 있음.

`FnOnce`: closure 안으로 ownership 넘어옴.

`Fn`:  borrows values from the environment immutably

`FnMut`: borrows values from the environment mutably

위의 equal_to_x 는 `Fn` Trait 을 구현함. 이를 아래와 같은 코드로 변경하여 `FnOnce` 로 변경가능 (feat. `move` keyword)

```
fn main() {
    let x = vec![1, 2, 3];

    let equal_to_x = move |z| z == x;

    println!("can't use x here: {:?}", x);

    let y = vec![1, 2, 3];

    assert!(equal_to_x(y));
}
```

## Iterators

Rust 의 iterator 는 *lazy* 함. 아래는 iterator 를 선언하는 코드임. 선언할때 별짓 안함.

```
let v1 = vec![1, 2, 3];

let v1_iter = v1.iter();
```

모든 iterator 는 `Iterator` trait 를 구현함. `Iterator` trait 는 다음과 같음

```
trait Iterator {
    type Item;

    fn next(&mut self) -> Option<Self::Item>;

    // methods with default implementations elided
}
```

처음 보는 문법이 등장함. `type Item` 과  `Self::Item` 인데 이 둘은 associated type 을 정의함. chapter 19 에서 자세히 다룸.

일단은 Item 을 우리가 정의해줘야하며, 그게 next 메소드 리턴 값의 타입이 된다는 것만 알아두자.

vector 의 `iter` method 는 불변 참조를, `into_iter` 는 ownership 을 `iter_mut` 은 가변 참조를 next method 에서 리턴함.

next 를 호출하는 메서드들을 consuming adapters 라고 함. 말 그대로 iterator 를 소비해버리기 때문 (=ownership 를 가져가버림.)

```
#[test]
fn iterator_sum() {
    let v1 = vec![1, 2, 3];

    let v1_iter = v1.iter();

    let total: i32 = v1_iter.sum();

    assert_eq!(total, 6);
}
```

sum 이 consuming adapter 이고, sum method 호출 후에는 v1_iter 의 life 는 끝남.

또 다른 예로 map 이 있음.

```
let v1: Vec<i32> = vec![1, 2, 3];

let v2: Vec<_> = v1.iter().map(|x| x + 1).collect();

assert_eq!(v2, vec![2, 3, 4]);
```

custom iterator 를 만들어 보자.

```
struct Counter {
    count: u32,
}

impl Counter {
    fn new() -> Counter {
        Counter { count: 0 }
    }
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        self.count += 1;

        if self.count < 6 {
            Some(self.count)
        } else {
            None
        }
    }
}
```

iterator 를 쓰는게 for 문 도는 것보다 속도가 빠름. loop unrolling 이 가능하기 때문인듯

</details>